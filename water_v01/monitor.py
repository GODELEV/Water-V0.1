import pygame
import sys
import math

class LiveMonitor:
    def __init__(self, width=900, height=450, title='Water v0.1 Training Monitor'):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.width = width
        self.height = height
        self.font = pygame.font.SysFont('Arial', 20)
        self.small_font = pygame.font.SysFont('Arial', 14)
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.epoch = 0
        self.dataset = ''
        self.running = True

    def update(self, train_loss, val_loss, train_acc, val_acc, epoch, dataset):
        if any(math.isnan(x) or math.isinf(x) for x in [train_loss, val_loss, train_acc, val_acc]):
            return
        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)
        self.train_acc_history.append(train_acc)
        self.val_acc_history.append(val_acc)
        self.epoch = epoch
        self.dataset = dataset
        self._draw()

    def _draw(self):
        self.screen.fill((30, 30, 30))
        graph_height = self.height - 100
        graph_width = (self.width - 80) // 2
        left1 = 30
        left2 = left1 + graph_width + 20
        top = 30
        # Draw axes for both graphs
        for left in [left1, left2]:
            pygame.draw.line(self.screen, (180, 180, 180), (left, top), (left, graph_height + top), 2)
            pygame.draw.line(self.screen, (180, 180, 180), (left, graph_height + top), (left + graph_width, graph_height + top), 2)
            for i in range(5):
                y = top + i * (graph_height // 4)
                pygame.draw.line(self.screen, (60, 60, 60), (left, y), (left + graph_width, y), 1)
            for i in range(5):
                x = left + i * (graph_width // 4)
                pygame.draw.line(self.screen, (60, 60, 60), (x, top), (x, graph_height + top), 1)
        # Loss graph (left)
        if self.train_loss_history:
            max_loss = max(self.train_loss_history + self.val_loss_history)
            min_loss = min(self.train_loss_history + self.val_loss_history)
            def scale_loss(val):
                return graph_height + top - int((val - min_loss) / (max_loss - min_loss + 1e-8) * (graph_height - 40))
            for history, color in [(self.train_loss_history, (200,0,0)), (self.val_loss_history, (0,200,200))]:
                points = [
                    (left1 + i * graph_width // max(1, len(history)-1), scale_loss(l))
                    for i, l in enumerate(history)
                ]
                if len(points) > 1:
                    pygame.draw.aalines(self.screen, color, False, points)
                for pt in points:
                    pygame.draw.circle(self.screen, color, pt, 2)
            # Y axis labels
            for i in range(5):
                y = top + i * (graph_height // 4)
                val = max_loss - (max_loss - min_loss) * (i / 4)
                label = self.small_font.render(f"{val:.4f}", True, (200,200,200))
                self.screen.blit(label, (left1-28, y-8))
            # X axis labels
            for i in range(5):
                x = left1 + i * (graph_width // 4)
                idx = int(i * (len(self.train_loss_history)-1) / 4) if len(self.train_loss_history) > 1 else 0
                label = self.small_font.render(f"{idx+1}", True, (200,200,200))
                self.screen.blit(label, (x-8, graph_height + top + 8))
            title = self.font.render("Loss", True, (255,255,255))
            self.screen.blit(title, (left1 + graph_width//2 - 30, 5))
        # Accuracy graph (right)
        if self.train_acc_history:
            max_acc = max(self.train_acc_history + self.val_acc_history)
            min_acc = min(self.train_acc_history + self.val_acc_history)
            def scale_acc(val):
                return graph_height + top - int((val - min_acc) / (max_acc - min_acc + 1e-8) * (graph_height - 40))
            for history, color in [(self.train_acc_history, (200,0,0)), (self.val_acc_history, (0,200,200))]:
                points = [
                    (left2 + i * graph_width // max(1, len(history)-1), scale_acc(a))
                    for i, a in enumerate(history)
                ]
                if len(points) > 1:
                    pygame.draw.aalines(self.screen, color, False, points)
                for pt in points:
                    pygame.draw.circle(self.screen, color, pt, 2)
            # Y axis labels
            for i in range(5):
                y = top + i * (graph_height // 4)
                val = max_acc - (max_acc - min_acc) * (i / 4)
                label = self.small_font.render(f"{val:.4f}", True, (200,200,200))
                self.screen.blit(label, (left2-28, y-8))
            # X axis labels
            for i in range(5):
                x = left2 + i * (graph_width // 4)
                idx = int(i * (len(self.train_acc_history)-1) / 4) if len(self.train_acc_history) > 1 else 0
                label = self.small_font.render(f"{idx+1}", True, (200,200,200))
                self.screen.blit(label, (x-8, graph_height + top + 8))
            title = self.font.render("Accuracy", True, (255,255,255))
            self.screen.blit(title, (left2 + graph_width//2 - 50, 5))
        # Draw text
        if self.train_loss_history:
            loss_text = self.font.render(f"Train Loss: {self.train_loss_history[-1]:.4f}  Val Loss: {self.val_loss_history[-1]:.4f}", True, (255,255,255))
        else:
            loss_text = self.font.render("Loss: N/A", True, (255,255,255))
        if self.train_acc_history:
            acc_text = self.font.render(f"Train Acc: {self.train_acc_history[-1]:.4f}  Val Acc: {self.val_acc_history[-1]:.4f}", True, (255,255,255))
        else:
            acc_text = self.font.render("Accuracy: N/A", True, (255,255,255))
        epoch_text = self.font.render(f"Epoch: {self.epoch}", True, (255,255,255))
        dataset_text = self.font.render(f"Dataset: {self.dataset}", True, (255,255,255))
        self.screen.blit(loss_text, (30, self.height-80))
        self.screen.blit(acc_text, (30, self.height-50))
        self.screen.blit(epoch_text, (self.width//2 - 60, self.height-80))
        self.screen.blit(dataset_text, (self.width//2 - 60, self.height-50))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                sys.exit()

    def close(self):
        pygame.quit() 